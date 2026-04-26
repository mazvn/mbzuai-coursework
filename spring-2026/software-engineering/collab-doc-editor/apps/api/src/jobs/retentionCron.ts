import cron from "node-cron";
import { retentionService } from "../integrations/retentionService";

// schedules daily retention cleanup.
// default schedule: every day at 02:00 server time.
// cron format: "minute hour day-of-month month day-of-week"
// 0 2 * * *  -> 02:00 every day

export function startRetentionCron() {
  // Avoid running in test environment
  if (process.env.NODE_ENV === "test") {
    return;
  }

  cron.schedule("0 2 * * *", async () => {
    try {
      const result = await retentionService.enforceAll();

      // eslint-disable-next-line no-console
      console.log("[RetentionCron] Retention cleanup completed:", result);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[RetentionCron] Retention cleanup failed:", err);
    }
  });

  // eslint-disable-next-line no-console
  console.log("[RetentionCron] Scheduled daily retention job at 02:00");
}