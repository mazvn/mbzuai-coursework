import { Router } from "express";
import authMiddleware from "../../middleware/authMiddleware";
import { inviteController } from "./inviteController";
import { requireDocumentRole } from "../../middleware/docRoleMiddleware";

const router = Router();

// organization invites

// preview org invite:
// GET /invites/preview?token=
// public so signup/login screens can show org info before auth.
router.get("/preview", inviteController.preview);

// accept org invite:
// POST /invites/accept-org
// requires login so the invite is bound to the authenticated user.
router.post("/accept-org", authMiddleware, inviteController.accept);

// list org users for invite picker:
// GET /invites/org-users?q=
router.get("/org-users", authMiddleware, inviteController.listOrgUsers);

// invite an org member to a document (direct permission grant):
// POST /invites/documents/:id
// body: { userId: string, message?: string }
// owner-only.
router.post(
  "/documents/:id",
  authMiddleware,
  requireDocumentRole(["Owner"]),
  inviteController.inviteToDocument
);

// document email invites (token-based)

// accept document invite via token:
// POST /invites/accept-document
// body: { token: string }
router.post("/accept-document", authMiddleware, inviteController.acceptInvite);

export default router;