import { Router } from "express";
import { jobController } from "./jobController";

const router = Router();

// POST /jobs/run
// body: { jobId, operation, selectedText, parameters? }
router.post("/run", jobController.run);
router.post("/stream", jobController.stream);

export default router;
