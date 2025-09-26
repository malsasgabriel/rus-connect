import * as React from 'react'

export function Table({ className, ...props }: React.HTMLAttributes<HTMLTableElement>) {
  return <table className={`w-full caption-bottom text-sm ${className || ''}`} {...props} />
}

export function TableHeader({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <thead className={className} {...props} />
}

export function TableBody({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={className} {...props} />
}

export function TableFooter({ className, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tfoot className={`bg-gray-800 font-medium ${className || ''}`} {...props} />
}

export function TableRow({ className, ...props }: React.HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={`border-b border-gray-700 transition-colors ${className || ''}`} {...props} />
}

export function TableHead({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={`h-10 px-2 text-left align-middle font-medium text-gray-300 ${className || ''}`}
      {...props}
    />
  )
}

export function TableCell({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={`p-2 align-middle ${className || ''}`} {...props} />
}

export function TableCaption({ className, ...props }: React.HTMLAttributes<HTMLTableCaptionElement>) {
  return <caption className={`mt-4 text-sm text-gray-400 ${className || ''}`} {...props} />
}
