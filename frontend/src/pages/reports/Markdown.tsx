import { Fragment, type ReactNode } from 'react'

/** **bold** inside a line, as React nodes (no HTML is injected). */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') ? <b key={i}>{part.slice(2, -2)}</b> : <Fragment key={i}>{part}</Fragment>)
}

/** Enough Markdown for report previews: headings, bullet and numbered lists, paragraphs, bold. */
export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = []
  let list: { ordered: boolean; items: string[] } | null = null
  const flush = () => {
    if (!list) return
    const Tag = list.ordered ? 'ol' : 'ul'
    blocks.push(<Tag key={blocks.length} className={`${list.ordered ? 'list-decimal' : 'list-disc'} pl-5 space-y-0.5`}>{list.items.map((t, i) => <li key={i}>{inline(t)}</li>)}</Tag>)
    list = null
  }
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()
    const h = /^(#{1,4})\s+(.*)$/.exec(line)
    const ul = /^\s*[-*]\s+(.*)$/.exec(line)
    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (h) { flush(); const size = ['text-lg', 'text-base', 'text-sm', 'text-sm'][h[1].length - 1]; blocks.push(<p key={blocks.length} className={`${size} font-semibold mt-3`}>{inline(h[2])}</p>) }
    else if (ul || ol) { const ordered = !!ol; if (!list || list.ordered !== ordered) { flush(); list = { ordered, items: [] } } list.items.push((ul ?? ol)![1]) }
    else if (!line.trim()) flush()
    else { flush(); blocks.push(<p key={blocks.length}>{inline(line)}</p>) }
  }
  flush()
  return <div className="space-y-2 text-sm leading-relaxed">{blocks}</div>
}
