"use client"

import {useState} from "react"
import {useRouter} from "next/navigation"
import {useForm} from "react-hook-form"
import {zodResolver} from "@hookform/resolvers/zod"
import {
    rosterGenerationSchema,
    type RosterGenerationFormData,
} from "@/lib/validations"
import {generateRoster} from "@/lib/api"
import {Button} from "@/components/ui/button"
import {Input} from "@/components/ui/input"
import {Label} from "@/components/ui/label"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import {Switch} from "@/components/ui/switch"
import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card"
import {Alert, AlertDescription} from "@/components/ui/alert"
import {Loader2} from "lucide-react"

interface GenerationFormProps {
    lastInputs?: {
        main_officers: string
        gl_counters: string
        handwritten_counters: string
        ot_counters: string
        ro_ra_officers: string
        sos_timings: string
        beam_width: number
    }
}

export function GenerationForm({lastInputs}: GenerationFormProps) {
    const router = useRouter()
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const {
        register,
        handleSubmit,
        formState: {errors},
        setValue,
        watch,
    } = useForm<RosterGenerationFormData>({
        resolver: zodResolver(rosterGenerationSchema),
        defaultValues: {
            mode: "arrival",
            main_officers_reported: lastInputs?.main_officers || "",
            report_gl_counters: lastInputs?.gl_counters || "",
            handwritten_counters: lastInputs?.handwritten_counters || "",
            ot_counters: lastInputs?.ot_counters || "",
            ro_ra_officers: lastInputs?.ro_ra_officers || "",
            sos_timings: lastInputs?.sos_timings || "",
            beam_width: lastInputs?.beam_width || 20,
            save_to_history: true,
        },
    })

    const mode = watch("mode")
    const saveToHistory = watch("save_to_history")

    const onSubmit = async (data: RosterGenerationFormData) => {
        setIsLoading(true)
        setError(null)

        try {
            const response = await generateRoster(data)

            // Store result in sessionStorage for display page
            sessionStorage.setItem("rosterResult", JSON.stringify(response))

            router.push("/generate/result")
        } catch (err: any) {
            setError(err.message || "Failed to generate roster")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>Generate Officer Roster</CardTitle>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                    {/* Operation Mode */}
                    <div className="space-y-2">
                        <Label>Operation Mode</Label>
                        <Select
                            value={mode}
                            onValueChange={(value) =>
                                setValue("mode", value as "arrival" | "departure")
                            }
                        >
                            <SelectTrigger>
                                <SelectValue/>
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="arrival">Arrival</SelectItem>
                                <SelectItem value="departure">Departure</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Main Officers */}
                    <div className="space-y-2">
                        <Label htmlFor="main_officers_reported">
                            Main Officers Reported *
                        </Label>
                        <Input
                            id="main_officers_reported"
                            placeholder="e.g., 1-18 or 1,3,5-10"
                            {...register("main_officers_reported")}
                        />
                        {errors.main_officers_reported && (
                            <p className="text-sm text-red-500">
                                {errors.main_officers_reported.message}
                            </p>
                        )}
                    </div>

                    {/* Ground Level Counters */}
                    <div className="space-y-2">
                        <Label htmlFor="report_gl_counters">Ground Level Counters</Label>
                        <Input
                            id="report_gl_counters"
                            placeholder="e.g., 4AC1, 8AC11, 12AC21"
                            {...register("report_gl_counters")}
                        />
                    </div>

                    {/* Handwritten Counters */}
                    <div className="space-y-2">
                        <Label htmlFor="handwritten_counters">
                            Handwritten/Takeover Counters
                        </Label>
                        <Input
                            id="handwritten_counters"
                            placeholder="e.g., 3AC12, 5AC13"
                            {...register("handwritten_counters")}
                        />
                    </div>

                    {/* OT Counters */}
                    <div className="space-y-2">
                        <Label htmlFor="ot_counters">OT Officer Counters</Label>
                        <Input
                            id="ot_counters"
                            placeholder="e.g., 2, 20, 40"
                            {...register("ot_counters")}
                        />
                    </div>

                    {/* RO/RA Officers */}
                    <div className="space-y-2">
                        <Label htmlFor="ro_ra_officers">
                            RO/RA Officers (Late/Early Adjustments)
                        </Label>
                        <Input
                            id="ro_ra_officers"
                            placeholder="e.g., 3RO2100, 11RO1700"
                            {...register("ro_ra_officers")}
                        />
                    </div>

                    {/* SOS Timings */}
                    <div className="space-y-2">
                        <Label htmlFor="sos_timings">SOS Officer Timings</Label>
                        <Input
                            id="sos_timings"
                            placeholder="e.g., (AC22)1000-1300, 2000-2200"
                            {...register("sos_timings")}
                        />
                    </div>

                    {/* Beam Width */}
                    <div className="space-y-2">
                        <Label htmlFor="beam_width">
                            Beam Width (Optimization Precision)
                        </Label>
                        <Input
                            id="beam_width"
                            type="number"
                            min={1}
                            max={100}
                            {...register("beam_width", {valueAsNumber: true})}
                        />
                        {errors.beam_width && (
                            <p className="text-sm text-red-500">
                                {errors.beam_width.message}
                            </p>
                        )}
                        <p className="text-sm text-gray-500">
                            Higher values = better optimization, slower generation
                        </p>
                    </div>

                    {/* Save to History */}
                    <div className="flex items-center space-x-2">
                        <Switch
                            id="save_to_history"
                            checked={saveToHistory}
                            onCheckedChange={(checked) =>
                                setValue("save_to_history", checked)
                            }
                        />
                        <Label htmlFor="save_to_history">Save to history</Label>
                    </div>

                    {/* Error Display */}
                    {error && (
                        <Alert variant="destructive">
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {/* Submit Button */}
                    <Button type="submit" className="w-full" disabled={isLoading}>
                        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin"/>}
                        {isLoading ? "Generating..." : "Generate Roster"}
                    </Button>
                </form>
            </CardContent>
        </Card>
    )
}