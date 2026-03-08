import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath, remarkGfm]}
      rehypePlugins={[rehypeKatex, rehypeHighlight]}
      components={{
        pre({ children }) {
          return <pre className="rounded-lg overflow-x-auto text-sm my-2">{children}</pre>;
        },
        code({ className, children, ...props }) {
          const isInline = !className;
          if (isInline) {
            return (
              <code className="bg-gray-200 text-gray-800 px-1.5 py-0.5 rounded text-xs font-mono" {...props}>
                {children}
              </code>
            );
          }
          return <code className={className} {...props}>{children}</code>;
        },
        table({ children }) {
          return (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full text-sm border-collapse border border-gray-300">{children}</table>
            </div>
          );
        },
        th({ children }) {
          return <th className="border border-gray-300 px-3 py-1.5 bg-gray-100 text-left font-medium">{children}</th>;
        },
        td({ children }) {
          return <td className="border border-gray-300 px-3 py-1.5">{children}</td>;
        },
        p({ children }) {
          return <p className="my-1.5 leading-relaxed">{children}</p>;
        },
        ul({ children }) {
          return <ul className="list-disc pl-5 my-1.5 space-y-0.5">{children}</ul>;
        },
        ol({ children }) {
          return <ol className="list-decimal pl-5 my-1.5 space-y-0.5">{children}</ol>;
        },
        blockquote({ children }) {
          return <blockquote className="border-l-3 border-blue-400 pl-3 my-2 text-gray-600 italic">{children}</blockquote>;
        },
        h3({ children }) {
          return <h3 className="font-semibold text-base mt-3 mb-1">{children}</h3>;
        },
        h4({ children }) {
          return <h4 className="font-medium text-sm mt-2 mb-1">{children}</h4>;
        },
        strong({ children }) {
          return <strong className="font-semibold text-gray-900">{children}</strong>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}
