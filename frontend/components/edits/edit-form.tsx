"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import {
    editCreationSchema,
    type EditCreationFormData,
} from "@/lib/validations"
import { createRosterEdit } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2 } from "lucide-react"

export function EditForm() {
    const router = useRouter()
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState<string | null>(null)

    const {
        register,
        handleSubmit,
        formState: { errors },
        setValue,
        watch,
        reset,
    } = useForm<EditCreationFormData>({
        resolver: zodResolver(editCreationSchema),
        defaultValues: {
            edit_type: "delete",
            slot_start: 0,
            slot_end: 0,
        },
    })

    const editType = watch("edit_type")

    const onSubmit = async (data: EditCreationFormData) => {
        setIsLoading(true)
        setError(null)
        setSuccess(null)

        try {
            const response = await createRosterEdit(data)
            setSuccess(response.message)
            reset()
            router.refresh() // Refresh server components
        } catch (err: never) {
            setError(err.message || "Failed to create edit")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle>Create Roster Edit</CardTitle>
            </CardHeader>
            <CardContent>
                <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
                    {/* Edit Type */}
                    <div className="space-y-2">
                        <Label>Edit Type</Label>
                        <Select
                            value={editType}
                            onValueChange={(value) =>
                                setValue("edit_type", value as "delete" | "swap" | "add")
                            }
                        >
                            <SelectTrigger>
                                <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="delete">Delete</SelectItem>
                                <SelectItem value="swap">Swap</SelectItem>
                                <SelectItem value="add">Add</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Officer ID */}
                    <div className="space-y-2">
                        <Label htmlFor="officer_id">Officer ID *</Label>
                        <Input
                            id="officer_id"
                            placeholder="e.g., M1"
                            {...register("officer_id")}
                        />
                        {errors.officer_id && (
                            <p className="text-sm text-red-500">
                                {errors.officer_id.message}
                            </p>
                        )}
                    </div>

                    {/* Officer ID 2 (for swap) */}
                    {editType === "swap" && (
                        <div className="space-y-2">
                            <Label htmlFor="officer_id_2">Second Officer ID</Label>
                            <Input
                                id="officer_id_2"
                                placeholder="e.g., M2"
                                {...register("officer_id_2")}
                            />
                        </div>
                    )}

                    {/* Counter Number (for add) */}
                    {editType === "add" && (
                        <div className="space-y-2">
                            <Label htmlFor="counter_no">Counter Number</Label>
                            <Input
                                id="counter_no"
                                type="number"
                                {...register("counter_no", { valueAsNumber: true })}
                            />
                        </div>
                    )}

                    {/* Slot Range */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="slot_start">Start Slot (0-47)</Label>
                            <Input
                                id="slot_start"
                                type="number"
                                min={0}
                                max={47}
                                {...register("slot_start", { valueAsNumber: true })}
                            />
                            {errors.slot_start && (
                                <p className="text-sm text-red-500">
                                    {errors.slot_start.message}
                                </p>
                            )}
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="slot_end">End Slot (0-47)</Label>
                            <Input
                                id="slot_end"
                                type="number"
                                min={0}
                                max={47}
                                {...register("slot_end", { valueAsNumber: true })}
                            />
                            {errors.slot_end && (
                                <p className="text-sm text-red-500">
                                    {errors.slot_end.message}
                                </p>
                            )}
                        </div>
                    </div>

                    {/* Time Range */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="time_start">Start Time (HHMM)</Label>
                            <Input
                                id="time_start"
                                placeholder="e.g., 1000"
                                maxLength={4}
                                {...register("time_start")}
                            />
                            {errors.time_start && (
                                <p className="text-sm text-red-500">
                                    {errors.time_start.message}
                                </p>
                            )}
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="time_end">End Time (HHMM)</Label>
                            <Input
                                id="time_end"
                                placeholder="e.g., 1130"
                                maxLength={4}
                                {...register("time_end")}
                            />
                            {errors.time_end && (
                                <p className="text-sm text-red-500">
                                    {errors.time_end.message}
                                </p>
                            )}
                        </div>
                    </div>

                    {/* Notes */}
                    <div className="space-y-2">
                        <Label htmlFor="notes">Notes (Optional)</Label>
                        <Textarea id="notes" {...register("notes")} />
                    </div>

                    {/* Messages */}
                    {error && (
                        <Alert variant="destructive">
                            <AlertDescription>{error}</AlertDescription>
                        </Alert>
                    )}

                    {success && (
                        <Alert>
                            <AlertDescription>{success}</AlertDescription>
                        </Alert>
                    )}

                    {/* Submit */}
                    <Button type="submit" className="w-full" disabled={isLoading}>
                        {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                        {isLoading ? "Saving..." : "Create Edit"}
                    </Button>
                </form>
            </CardContent>
        </Card>
    )
}