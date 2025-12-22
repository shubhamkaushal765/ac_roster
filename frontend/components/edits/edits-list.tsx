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
import { RosterEdit } from "@/lib/types"
import { formatTimestamp } from "@/lib/utils"

interface EditsListProps {
    edits: RosterEdit[]
}

export function EditsList({ edits }: EditsListProps) {
    if (edits.length === 0) {
        return (
            <Card>
                <CardContent className="pt-6">
            <p className="text-center text-gray-500">No edits found</p>
        </CardContent>
        </Card>
    )
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>Recent Edits</CardTitle>
    </CardHeader>
    <CardContent>
    <div className="overflow-auto">
        <Table>
            <TableHeader>
                <TableRow>
                    <TableHead>Timestamp</TableHead>
        <TableHead>Type</TableHead>
        <TableHead>Officer(s)</TableHead>
        <TableHead>Slots</TableHead>
        <TableHead>Time Range</TableHead>
    <TableHead>Notes</TableHead>
    </TableRow>
    </TableHeader>
    <TableBody>
    {edits.map((edit) => (
            <TableRow key={edit.id}>
                <TableCell>{formatTimestamp(edit.timestamp)}</TableCell>
    <TableCell>
    <Badge
        variant={
            edit.edit_type === "delete"
                ? "destructive"
                : edit.edit_type === "swap"
                    ? "default"
                    : "secondary"
        }
        >
        {edit.edit_type}
        </Badge>
        </TableCell>
        <TableCell>
        {edit.officer_id}
    {edit.officer_id_2 && ` ↔ ${edit.officer_id_2}`}
    </TableCell>
    <TableCell>
    {edit.slot_start} - {edit.slot_end}
    </TableCell>
    <TableCell>
    {edit.time_start} - {edit.time_end}
    </TableCell>
    <TableCell className="max-w-xs truncate">
        {edit.notes || "-"}
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