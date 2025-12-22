import { GenerationForm } from "@/components/roster/generation-form"
import { getLastInputs } from "@/lib/api"

export default async function GeneratePage() {
    // SSR: Fetch last inputs to pre-fill form
    let lastInputs
    try {
        const response = await getLastInputs()
        lastInputs = response.data
    } catch (error) {
        lastInputs = undefined
    }

    return (
        <div className="max-w-2xl mx-auto">
            <h1 className="text-3xl font-bold mb-6">Generate Roster</h1>
            <GenerationForm lastInputs={lastInputs} />
        </div>
    )
}