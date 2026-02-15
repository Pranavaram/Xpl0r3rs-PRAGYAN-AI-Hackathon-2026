/**
 * Extract plain text from a PDF file (ArrayBuffer) using pdfjs-dist.
 * Returns combined text from all pages.
 */
export async function extractTextFromPdf(arrayBuffer: ArrayBuffer): Promise<string> {
  const pdfjsLib = await import('pdfjs-dist');
  const workerUrl = await import('pdfjs-dist/build/pdf.worker.min.mjs?url').then((m) => m.default);
  const GlobalWorkerOptions = (pdfjsLib as unknown as { GlobalWorkerOptions?: { workerSrc: string } }).GlobalWorkerOptions;
  if (GlobalWorkerOptions && workerUrl) GlobalWorkerOptions.workerSrc = workerUrl;

  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  const pdf = await loadingTask.promise;
  const numPages = pdf.numPages;
  const parts: string[] = [];
  for (let i = 1; i <= numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const items = content.items as Array<{ str?: string }>;
    const strings = items.map((item) => item.str ?? '').filter(Boolean);
    parts.push(strings.join(' '));
  }
  return parts.join('\n\n');
}
