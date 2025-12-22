"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { RosterDisplay } from "@/components/roster/roster-display"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { RosterGenerationResponse } from "@/lib/types"

export default function ResultPage() {
    const router = useRouter()
    const [result, setResult] = useState<RosterGenerationResponse | null>(null)

    useEffect(() => {
        // Retrieve result from sessionStorage (set in form submission)
        const storedResult = sessionStorage.getItem("rosterResult")
        if (storedResult) {
            setResult(JSON.parse(storedResult))
            sessionStorage.removeItem("rosterResult")
        } else {
            router.push("/generate")
        }
    }, [router])

    if (!result) {
        return (
            <Card>
                <CardContent className="pt-6">
                    <p className="text-center">Loading result...</p>
                </CardContent>
            </Card>
        )
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Roster Generated</h1>
                <Button onClick={() => router.push("/generate")} variant="outline">
                    Generate Another
                </Button>
            </div>

            <RosterDisplay result={result} />

            {/* Statistics */}
            {result.statistics && (
                <Card>
                    <CardContent className="pt-6 space-y-4">
                        <div>
                            <h3 className="font-semibold mb-2">
                                Before SOS Optimization:
                            </h3>
                            <pre className="text-xs bg-gray-50 p-4 rounded overflow-auto">
                {result.statistics.stats1}
              </pre>
                        </div>
                        <div>
                            <h3 className="font-semibold mb-2">After SOS Optimization:</h3>
                            <pre className="text-xs bg-gray-50 p-4 rounded overflow-auto">
                {result.statistics.stats2}
              </pre>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    )
}