import { Router } from "express";
import { authController } from "./authController";
import authMiddleware from "../../middleware/authMiddleware";
import { validateRequest } from "../../middleware/validateRequest";

import {
  loginRequestSchema,
  signupRequestSchema,
  inviteSignupRequestSchema,
  ownerSignupRequestSchema,
  createOrganizationRequestSchema,
} from "@repo/contracts";

const router = Router();

// POST /auth/login
router.post(
  "/login",
  validateRequest({ body: loginRequestSchema }),
  authController.login
);

// POST /auth/signup
// create standalone user (no organization)
router.post(
  "/signup",
  validateRequest({ body: signupRequestSchema }),
  authController.signup
);

// POST /auth/signup-owner
// create account + organization (OrgOwner)
router.post(
  "/signup-owner",
  validateRequest({ body: ownerSignupRequestSchema }),
  authController.signupOwner
);

// POST /auth/signup-invite
// create account + accept org invite
router.post(
  "/signup-invite",
  validateRequest({ body: inviteSignupRequestSchema }),
  authController.signupInvite
);

// POST /auth/create-organization
// logged-in user creates a new organization and becomes OrgOwner
router.post(
  "/create-organization",
  authMiddleware,
  validateRequest({ body: createOrganizationRequestSchema }),
  authController.createOrganization
);

// GET /auth/me
router.get(
  "/me",
  authMiddleware,
  authController.me
);

// GET /auth/organizations
// list organizations the current user belongs to, with document summaries.
router.get(
  "/organizations",
  authMiddleware,
  authController.listOrganizations
);

// POST /auth/switch-organization
// switch active organization context for the current session.
router.post(
  "/switch-organization",
  authMiddleware,
  authController.switchOrganization
);

// DELETE /auth/organizations/:orgId/leave
// leave an organization as a non-owner member/admin.
router.delete(
  "/organizations/:orgId/leave",
  authMiddleware,
  authController.leaveOrganization
);

// POST /auth/refresh
// uses HttpOnly refresh token cookie to silently issue a new access token.
router.post(
  "/refresh",
  authController.refresh
);

// DELETE /auth/account
// self-delete account
router.delete(
  "/account",
  authMiddleware,
  authController.deleteAccount
);

// POST /auth/logout
router.post(
  "/logout",
  authMiddleware,
  authController.logout
);

export default router;
