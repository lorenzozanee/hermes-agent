import {
  Button,
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  Textarea
} from '@hermes/plugin-sdk'
import { useEffect, useState } from 'react'

import { FIELD_LABEL, useKanban } from './ui'

export const hasCompletionResult = (result?: null | string): boolean => Boolean(result?.trim())

export function CompletionSummaryDialog({
  description,
  onCancel,
  onSubmit,
  open,
  pending = false
}: {
  description?: string
  onCancel: () => void
  onSubmit: (summary: string) => void
  open: boolean
  pending?: boolean
}) {
  const k = useKanban()
  const [summary, setSummary] = useState('')
  const trimmed = summary.trim()

  useEffect(() => {
    if (open) {
      setSummary('')
    }
  }, [open])

  return (
    <Dialog onOpenChange={next => !next && !pending && onCancel()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{k.completionPrompt}</DialogTitle>
          <DialogDescription>{description ?? k.completionDescription}</DialogDescription>
        </DialogHeader>
        <label className="flex flex-col gap-1">
          <span className={FIELD_LABEL}>{k.result}</span>
          <Textarea
            aria-label={k.result}
            autoFocus
            onChange={event => setSummary(event.target.value)}
            placeholder={k.completionPlaceholder}
            value={summary}
          />
        </label>
        <DialogFooter>
          <Button disabled={pending} onClick={onCancel} variant="text">
            {k.cancel}
          </Button>
          <Button disabled={!trimmed || pending} onClick={() => onSubmit(trimmed)}>
            {k.completeTask}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
