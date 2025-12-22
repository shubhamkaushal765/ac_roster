import { getRosterEdits } from "@/lib/api"
import { EditForm } from "@/components/edits/edit-form"
import { EditsList } from "@/components/edits/edits-list"
import { RosterEdit } from "@/lib/types"

export default async function EditsPage() {
    // SSR: Fetch edits on server
    let edits: RosterEdit[]
    try {
        const response = await getRosterEdits(50)
        edits = response.data
    } catch (error) {
        edits = []
    }

    return (
        <div className="space-y-8">
            <h1 className="text-3xl font-bold">Roster Edits</h1>

            <div className="grid gap-8 lg:grid-cols-2">
                {/* Form (Client Component) */}
                <div>
                    <EditForm />
                </div>

                {/* List (Server Component) */}
                <div>
                    <EditsList edits={edits} />
                </div>
            </div>
        </div>
    )
}