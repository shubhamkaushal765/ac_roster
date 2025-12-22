import {z} from "zod"

export const rosterGenerationSchema = z.object({
    mode: z.enum(["arrival", "departure"]).default("arrival"),
    main_officers_reported: z
        .string()
        .min(1, "Officers list is required")
        .regex(
            /^[\d,\-\s]+$/,
            "Invalid format. Use ranges (1-18) or comma-separated (1,3,5)"
        ),
    report_gl_counters: z.string().default(""),
    handwritten_counters: z.string().default(""),
    ot_counters: z.string().default(""),
    ro_ra_officers: z.string().default(""),
    sos_timings: z.string().default(""),
    beam_width: z.coerce.number().min(1).max(100).default(20),
    save_to_history: z.boolean().default(true),
})

export const editCreationSchema = z
    .object({
        edit_type: z.enum(["delete", "swap", "add"]),
        officer_id: z.string().min(1, "Officer ID is required"),
        officer_id_2: z.string().optional(),
        counter_no: z.coerce.number().int().min(0).optional(),
        slot_start: z.coerce.number().int().min(0).max(47),
        slot_end: z.coerce.number().int().min(0).max(47),
        time_start: z
            .string()
            .regex(/^\d{4}$/, "Time must be in HHMM format (e.g., 1000)"),
        time_end: z
            .string()
            .regex(/^\d{4}$/, "Time must be in HHMM format (e.g., 1130)"),
        roster_history_id: z.coerce.number().int().positive().optional(),
        notes: z.string().optional(),
    })
    .refine((data) => data.slot_end >= data.slot_start, {
        message: "End slot must be greater than or equal to start slot",
        path: ["slot_end"],
    })

export type RosterGenerationFormData = z.infer<
    typeof rosterGenerationSchema
>
export type EditCreationFormData = z.infer<typeof editCreationSchema>