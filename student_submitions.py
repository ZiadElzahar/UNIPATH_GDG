"""Backward-compatible import shim for the new package layout."""

from src.unipath.portal.student_submissions import StudentRegistrationSystem


if __name__ == "__main__":
    system = StudentRegistrationSystem()
    system.run()