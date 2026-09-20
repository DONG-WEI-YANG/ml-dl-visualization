import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import App from '../App';

afterEach(() => {
  localStorage.clear();
  window.history.replaceState({}, '', '/');
  vi.unstubAllGlobals();
});

it('blocks protected page requests until password rotation and then uses the replacement token', async () => {
  window.history.replaceState({}, '', '/admin/users');
  localStorage.setItem('auth_token', 'initial-token');
  const user = { id: 1, username: 'admin', display_name: 'Admin', role: 'admin', semester: '115-1', must_change_password: true };
  const fetcher = vi.fn(async (url: string, init?: RequestInit) => {
    if (url.endsWith('/api/auth/me')) {
      const fresh = (init?.headers as Record<string, string>)?.Authorization === 'Bearer rotated-token';
      return Response.json({ ...user, must_change_password: !fresh });
    }
    if (url.endsWith('/api/auth/change-password')) return Response.json({ access_token: 'rotated-token', user: { ...user, must_change_password: false } });
    return Response.json([]);
  });
  vi.stubGlobal('fetch', fetcher);
  render(<App />);
  expect(await screen.findByRole('dialog', { name: '變更密碼' })).toBeInTheDocument();
  expect(fetcher.mock.calls.filter(([url]) => url.includes('/api/admin/'))).toHaveLength(0);
  fireEvent.change(screen.getByLabelText('舊密碼'), { target: { value: 'initial-password' } });
  fireEvent.change(screen.getByLabelText('新密碼（8 碼以上）'), { target: { value: 'changed-password' } });
  fireEvent.change(screen.getByLabelText('確認新密碼'), { target: { value: 'changed-password' } });
  fireEvent.click(screen.getByRole('button', { name: '確認變更' }));
  await waitFor(() => expect(screen.queryByRole('dialog', { name: '變更密碼' })).not.toBeInTheDocument());
  expect(localStorage.getItem('auth_token')).toBe('rotated-token');
  await waitFor(() => expect(fetcher.mock.calls.some(([url]) => url.includes('/api/admin/users'))).toBe(true));
  for (const [url, init] of fetcher.mock.calls) {
    if (url.includes('/api/admin/')) expect((init?.headers as Record<string, string>).Authorization).toBe('Bearer rotated-token');
  }
});
