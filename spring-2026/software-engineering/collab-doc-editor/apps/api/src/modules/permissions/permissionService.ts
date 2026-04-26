import type { DocumentRole } from "@repo/contracts";
import { documentRepo } from "../documents/documentRepo";
import { permissionRepo } from "./permissionRepo";
import { userRepo } from "../auth/userRepo";
import { prisma } from "../../lib/prisma";

// resolves a user's effective role on a document.
// hard gates:
// - Document must exist
// - User must exist
const ROLE_RANK: Record<DocumentRole, number> = {
  Viewer: 1,
  Commenter: 2,
  Editor: 3,
  Owner: 4,
};

export const permissionService = {
  async resolveEffectiveRole(params: {
    documentId: string;
    userId: string;
    linkToken?: string;
  }): Promise<DocumentRole | null> {
    const { documentId, userId, linkToken } = params;

    const document = await documentRepo.findById(documentId);
    if (!document) return null;

    const user = await userRepo.findById(userId);
    if (!user) return null;

    const membership = await prisma.organizationMember.findUnique({
      where: {
        orgId_userId: {
          orgId: document.orgId,
          userId,
        },
      },
      select: { id: true },
    });

    if (!membership) return null;

    // Document owner always has Owner role
    if (document.ownerId === userId) {
      return "Owner";
    }

    // Explicit user permission
    const userPerm = await permissionRepo.findByUser(document.id, userId);
    if (userPerm) {
      return userPerm.role as DocumentRole;
    }

    // Link-based permission: only after org-membership gate passes
    if (linkToken) {
      const linkPerm = await permissionRepo.findByLink(document.id, linkToken);
      if (linkPerm) {
        return linkPerm.role as DocumentRole;
      }
    }

    return null;
  },

  hasRequiredRole(effectiveRole: DocumentRole | null, allowedRoles: DocumentRole[]): boolean {
    if (!effectiveRole) return false;
    return allowedRoles.includes(effectiveRole);
  },

  hasAtLeastRole(effectiveRole: DocumentRole | null, minimumRole: DocumentRole): boolean {
    if (!effectiveRole) return false;
    return ROLE_RANK[effectiveRole] >= ROLE_RANK[minimumRole];
  },

  isViewerOrAbove(effectiveRole: DocumentRole | null): boolean {
    return this.hasAtLeastRole(effectiveRole, "Viewer");
  },

  isCommenterOrAbove(effectiveRole: DocumentRole | null): boolean {
    return this.hasAtLeastRole(effectiveRole, "Commenter");
  },

  isEditorOrOwner(effectiveRole: DocumentRole | null): boolean {
    return this.hasAtLeastRole(effectiveRole, "Editor");
  },

  isOwner(effectiveRole: DocumentRole | null): boolean {
    return effectiveRole === "Owner";
  },
};