import {type ClassValue, clsx} from "clsx"
import {twMerge} from "tailwind-merge"
import {format} from "date-fns"

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs))
}

// Convert slot index (0-47) to time string (HHMM)
export function slotToTime(slot: number): string {
    const hours = Math.floor(slot / 2)
    const minutes = (slot % 2) * 30
    return `${hours.toString().padStart(2, "0")}${minutes
        .toString()
        .padStart(2, "0")}`
}

// Convert time string (HHMM) to slot index (0-47)
export function timeToSlot(time: string): number {
    const hours = parseInt(time.substring(0, 2), 10)
    const minutes = parseInt(time.substring(2, 4), 10)
    return hours * 2 + Math.floor(minutes / 30)
}

// Format timestamp for display
export function formatTimestamp(timestamp: string): string {
    return format(new Date(timestamp), "MMM dd, yyyy HH:mm")
}

// Format ISO date to readable format
export function formatDate(date: string): string {
    return format(new Date(date), "PPP")
}