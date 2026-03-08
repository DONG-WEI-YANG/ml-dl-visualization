import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center h-full p-8">
      <h1 className="text-6xl font-bold text-gray-300 mb-4">404</h1>
      <p className="text-lg text-gray-600 mb-6">找不到此頁面</p>
      <Link
        to="/"
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
      >
        返回首頁
      </Link>
    </div>
  );
}
