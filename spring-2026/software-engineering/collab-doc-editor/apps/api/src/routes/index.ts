import { Router } from "express";

import authRoutes from "../modules/auth/authRoutes";
import documentRoutes from "../modules/documents/documentRoutes";
import commentRoutes from "../modules/comments/commentRoutes";
import aiRoutes from "../modules/ai/aiRoutes";
import adminRoutes from "../modules/admin/adminRoutes";
import inviteRoutes from "../modules/invites/inviteRoutes";

const router = Router();

// auth routes
// mounted at /auth
router.use("/auth", authRoutes);

// org invite routes
// mounted at /org/invites
router.use("/org/invites", inviteRoutes);

// document routes
// mounted at /documents
router.use("/documents", documentRoutes);

// comment routes
// mounted at /documents/:id/comments
router.use("/documents/:id/comments", commentRoutes);

// AI routes
// mounted at /ai
router.use("/ai", aiRoutes);

// admin routes
// mounted at /admin
router.use("/admin", adminRoutes);
router.use("/invite", inviteRoutes);

export default router;