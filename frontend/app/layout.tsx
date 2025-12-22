import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Link from "next/link"
import { Button } from "@/components/ui/button"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
    title: "Officer Roster Optimization",
    description: "Optimize officer allocation on counters",
}

export default function RootLayout({
                                       children,
                                   }: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
        <body className={inter.className}>
        <div className="min-h-screen bg-gray-50">
            {/* Navigation */}
            <nav className="bg-white border-b">
                <div className="container mx-auto px-4 py-4">
                    <div className="flex items-center justify-between">
                        <Link href="/" className="text-xl font-bold">
                            Roster Optimization
                        </Link>
                        <div className="flex gap-4">
                            <Button variant="ghost" asChild>
                                <Link href="/generate">Generate</Link>
                            </Button>
                            <Button variant="ghost" asChild>
                                <Link href="/history">History</Link>
                            </Button>
                            <Button variant="ghost" asChild>
                                <Link href="/edits">Edits</Link>
                            </Button>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content */}
            <main className="container mx-auto px-4 py-8">{children}</main>
        </div>
        </body>
        </html>
    )
}