import { aiJobRepo } from "../modules/ai/aiJobRepo";
import { aiJobApplicationRepo } from "../modules/ai/aiJobApplicationRepo";
import { auditLogRepo } from "../modules/audit/auditLogRepo";
import { config } from "../config/env";

// retention policies:
// 1. AI interactions (jobs + applications)
// - Default: 30 days
// 2. Audit logs

const DEFAULT_AI_RETENTION_DAYS = 30;

function computeCutoff(days: number): Date {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - days);
  return cutoff;
}

export const retentionService = {
  // enforce AI interaction retention (30 days default)
  async enforceAIInteractionRetention(
    retentionDays: number = DEFAULT_AI_RETENTION_DAYS
  ) {
    const cutoff = computeCutoff(retentionDays);

    // Delete applications first (FK dependency)
    const appsDeleted = await aiJobApplicationRepo.deleteOlderThan(cutoff);

    // Then delete jobs
    const jobsDeleted = await aiJobRepo.deleteOlderThan(cutoff);

    return {
      type: "ai",
      retentionDays,
      cutoff: cutoff.toISOString(),
      aiJobApplicationsDeleted: appsDeleted.count,
      aiJobsDeleted: jobsDeleted.count,
    };
  },

  // enforce audit log retention (config-driven, default 90 days)
  async enforceAuditLogRetention(
    retentionDays: number = config.AUDIT_LOG_RETENTION_DAYS
  ) {
    const cutoff = computeCutoff(retentionDays);

    const deleted = await auditLogRepo.deleteOlderThan(cutoff);

    return {
      type: "audit",
      retentionDays,
      cutoff: cutoff.toISOString(),
      auditLogsDeleted: deleted.count,
    };
  },

  // run all retention policies together
  async enforceAll() {
    const aiResult = await this.enforceAIInteractionRetention();
    const auditResult = await this.enforceAuditLogRetention();

    return {
      timestamp: new Date().toISOString(),
      ai: aiResult,
      audit: auditResult,
    };
  },
};