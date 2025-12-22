import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import { RosterGenerationResponse } from "@/lib/types"
import { slotToTime } from "@/lib/utils"

interface RosterDisplayProps {
    result: RosterGenerationResponse
}

export function RosterDisplay({ result }: RosterDisplayProps) {
    const { data, officer_counts, optimization_penalty } = result

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-4">
                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">
                            Main Officers
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{officer_counts.main}</div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">
                            SOS Officers
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{officer_counts.sos}</div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">
                            OT Officers
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{officer_counts.ot}</div>
                    </CardContent>
                </Card>

                <Card>
                    <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium">
                            Total Officers
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{officer_counts.total}</div>
                    </CardContent>
                </Card>
            </div>

            {/* Optimization Penalty */}
            {optimization_penalty !== null && (
                <Card>
                    <CardHeader>
                        <CardTitle className="text-sm">Optimization Penalty</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">
                {optimization_penalty.toFixed(2)}
              </span>
                            <Badge
                                variant={optimization_penalty < 10 ? "default" : "destructive"}
                            >
                                {optimization_penalty < 10 ? "Good" : "Needs Improvement"}
                            </Badge>
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Officer Schedules */}
            <Card>
                <CardHeader>
                    <CardTitle>Officer Schedules</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="max-h-[400px] overflow-auto">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Officer ID</TableHead>
                                    <TableHead>Assigned Slots</TableHead>
                                    <TableHead>Time Range</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {Object.entries(data.officer_schedules).map(
                                    ([officerId, slots]) => {
                                        const slotArray = slots as number[]
                                        const firstSlot = Math.min(...slotArray)
                                        const lastSlot = Math.max(...slotArray)

                                        return (
                                            <TableRow key={officerId}>
                                                <TableCell className="font-medium">
                                                    {officerId}
                                                </TableCell>
                                                <TableCell>{slotArray.length} slots</TableCell>
                                                <TableCell>
                                                    {slotToTime(firstSlot)} - {slotToTime(lastSlot + 1)}
                                                </TableCell>
                                            </TableRow>
                                        )
                                    }
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </CardContent>
            </Card>

            {/* Counter Matrix Preview */}
            <Card>
                <CardHeader>
                    <CardTitle>Counter Coverage Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-500 mb-2">
                        {data.counter_matrix.length} counters × {data.counter_matrix[0]?.length || 0} time slots
                    </p>
                    <div className="text-xs text-gray-400">
                        Full matrix visualization can be added here
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}