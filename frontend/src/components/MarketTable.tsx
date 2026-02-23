
import React from 'react';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getSortedRowModel,
  SortingState,
} from '@tanstack/react-table';
import { MarketPair } from '../types';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './ui/table'; // Assuming shadcn/ui table components

interface MarketTableProps {
  data: MarketPair[];
}

export const MarketTable: React.FC<MarketTableProps> = ({ data }: MarketTableProps) => {
  const [sorting, setSorting] = React.useState<SortingState>([]);

  const columns: ColumnDef<MarketPair>[] = [
    {
      accessorKey: 'symbol',
      header: 'Symbol',
      cell: info => <div className="font-medium">{info.getValue<string>()}</div>,
    },
    {
      accessorKey: 'price',
      header: () => <div className="text-right">Price</div>,
      cell: info => <div className="text-right">{Number(info.getValue<number>()).toFixed(2)}</div>,
      sortingFn: 'alphanumeric',
    },
    {
      accessorKey: 'volume',
      header: () => <div className="text-right">Volume (24h)</div>,
      cell: info => <div className="text-right">{Number(info.getValue<number>()).toLocaleString()}</div>,
      sortingFn: 'alphanumeric',
    },
    {
      accessorKey: 'anomaly_score',
      header: () => <div className="text-right">Anomaly Score</div>,
      cell: info => {
        const score = Number(info.getValue<number>());
        let colorClass = 'text-gray-400';
        if (score > 70) colorClass = 'text-red-500';
        else if (score > 50) colorClass = 'text-orange-400';
        else if (score > 30) colorClass = 'text-yellow-400';
        else if (score > 0) colorClass = 'text-green-500';
        
        return (
          <div className={`text-right font-bold ${colorClass}`}>
            {score.toFixed(1)}
          </div>
        );
      },
      sortingFn: 'alphanumeric',
    },
    {
      accessorKey: 'last_update',
      header: () => <div className="text-right">Last Update</div>,
      cell: info => <div className="text-right">{new Date(Number(info.getValue<number>()) * 1000).toLocaleTimeString()}</div>,
      sortingFn: 'alphanumeric',
    },
  ];

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    state: {
      sorting,
    },
  });

  return (
    <div className="rounded-md border bg-gray-800 border-gray-700">
      <Table>
        <TableHeader>
          {table.getHeaderGroups().map((headerGroup) => (
            <TableRow key={headerGroup.id} className="hover:bg-gray-700">
              {headerGroup.headers.map((header) => (
                <TableHead key={header.id} className="text-gray-300">
                  {header.isPlaceholder
                    ? null
                    : (
                      <div
                        {...{
                          className: header.column.getCanSort()
                            ? 'cursor-pointer select-none'
                            : '',
                          onClick: header.column.getToggleSortingHandler(),
                        }}
                      >
                        {flexRender(
                          header.column.columnDef.header,
                          header.getContext()
                        )}
                        {{
                          asc: ' 🔼',
                          desc: ' 🔽',
                        }[header.column.getIsSorted() as string] ?? null}
                      </div>
                    )}
                </TableHead>
              ))}
            </TableRow>
          ))}
        </TableHeader>
        <TableBody>
          {table.getRowModel().rows?.length ? (
            table.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                data-state={row.getIsSelected() && 'selected'}
                className="hover:bg-gray-700"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="text-gray-200">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : (
            <TableRow>
              <TableCell colSpan={columns.length} className="h-24 text-center text-gray-400">
                No results.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  );
};
