import { getRosterHistory } from "@/lib/api"
import { HistoryTable } from "@/components/history/history-table"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { RosterHistoryItem } from "@/lib/types"

export default async function HistoryPage() {
    // SSR: Fetch history on server
    let history: RosterHistoryItem[]
    try {
        const response = await getRosterHistory(50)
        history = response.data
    } catch (error) {
        history = []
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Roster History</h1>
                <Button asChild>
                    <Link href="/generate">Generate New</Link>
                </Button>
            </div>

            <HistoryTable items={history} />
        </div>
    )
}