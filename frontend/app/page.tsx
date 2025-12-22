import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { getRosterHistory } from "@/lib/api"
import { HistoryTable } from "@/components/history/history-table"
import { RosterHistoryItem } from "@/lib/types"

export default async function HomePage() {
  // SSR: Fetch recent history on server
  let recentHistory: RosterHistoryItem[]
  try {
    const response = await getRosterHistory(5)
    recentHistory = response.data
  } catch (error) {
    recentHistory = []
  }

  return (
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold">Officer Roster Optimization</h1>
          <p className="text-gray-600 mt-2">
            Generate optimized officer schedules for counter allocation
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Generate Roster</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-4">
                Create a new optimized officer roster
              </p>
              <Button asChild className="w-full">
                <Link href="/generate">Start Generation</Link>
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">View History</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-4">
                Browse past roster generations
              </p>
              <Button asChild variant="outline" className="w-full">
                <Link href="/history">View All</Link>
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Manage Edits</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-gray-600 mb-4">
                Create and manage roster edits
              </p>
              <Button asChild variant="outline" className="w-full">
                <Link href="/edits">Manage Edits</Link>
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Recent History */}
        <div>
          <h2 className="text-2xl font-bold mb-4">Recent Generations</h2>
          <HistoryTable items={recentHistory} />
        </div>
      </div>
  )
}