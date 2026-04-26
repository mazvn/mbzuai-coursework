// document-level roles
// these apply per document and control document content access.
export const DOCUMENT_ROLES = [
  "Viewer",
  "Commenter",
  "Editor",
  "Owner",
] as const;

export type DocumentRole = (typeof DOCUMENT_ROLES)[number];

// organization-level roles
// these apply at the organization scope only.
export const ORG_ROLES = [
  "OrgAdmin",
  "OrgOwner",
] as const;

export type OrgRole = (typeof ORG_ROLES)[number] | null;

// utility helpers

export const isDocumentRole = (value: unknown): value is DocumentRole => {
  return typeof value === "string" &&
    (DOCUMENT_ROLES as readonly string[]).includes(value);
};

export const isOrgRole = (value: unknown): value is OrgRole => {
  if (value === null) return true;
  return typeof value === "string" &&
    (ORG_ROLES as readonly string[]).includes(value);
};