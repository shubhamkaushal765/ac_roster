import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { RosterHistoryItem } from "@/lib/types"
import { formatTimestamp } from "@/lib/utils"

interface HistoryTableProps {
    items: RosterHistoryItem[]
}

export function HistoryTable({ items }: HistoryTableProps) {
    if (items.length === 0) {
        return (
            <Card>
                <CardContent className="pt-6">
                    <p className="text-center text-gray-500">No history records found</p>
                </CardContent>
            </Card>
        )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>Roster Generation History</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="overflow-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Timestamp</TableHead>
                                <TableHead>Officers</TableHead>
                                <TableHead>Total</TableHead>
                                <TableHead>Beam Width</TableHead>
                                <TableHead>Penalty</TableHead>
                                <TableHead>Notes</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {items.map((item) => (
                                <TableRow key={item.id}>
                                    <TableCell className="font-medium">
                                        {formatTimestamp(item.timestamp)}
                                    </TableCell>
                                    <TableCell>
                                        <div className="text-sm">
                                            <div>Main: {item.main_officer_count ?? "N/A"}</div>
                                            <div className="text-gray-500">
                                                SOS: {item.sos_officer_count ?? 0} | OT:{" "}
                                                {item.ot_officer_count ?? 0}
                                            </div>
                                        </div>
                                    </TableCell>
                                    <TableCell>
                                        <Badge>{item.total_officer_count ?? "N/A"}</Badge>
                                    </TableCell>
                                    <TableCell>{item.beam_width}</TableCell>
                                    <TableCell>
                                        {typeof item.optimization_penalty === "number"
                                            ? item.optimization_penalty.toFixed(2)
                                            : "N/A"}
                                    </TableCell>
                                    <TableCell className="max-w-xs truncate">
                                        {item.notes || "-"}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    )
}