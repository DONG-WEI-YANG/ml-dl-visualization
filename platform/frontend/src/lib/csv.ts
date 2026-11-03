const BOM = new RegExp("^" + String.fromCharCode(0xfeff));
const HEADER_FIRST_CELLS = ["學號", "帳號", "username", "account", "id"];

/** Drop a BOM and a leading header row such as "學號,姓名,Email". */
export function stripHeader(text: string): string {
  const lines = text.replace(BOM, "").split(/\r?\n/);
  const first = (lines[0] ?? "").split(",")[0].trim().toLowerCase();
  if (HEADER_FIRST_CELLS.includes(first)) lines.shift();
  return lines.join("\n");
}
