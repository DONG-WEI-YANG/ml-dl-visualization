"""Client address resolution behind the Hugging Face reverse proxy."""


def client_ip(request) -> str:
    """Best-effort real client address.

    On HF Spaces uvicorn sees every request from the proxy, so the socket
    address is shared by all users. The proxy appends the address it saw to
    X-Forwarded-For; only that rightmost entry is trustworthy — entries to its
    left are whatever the client sent and could be forged to dodge limits.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else ""
