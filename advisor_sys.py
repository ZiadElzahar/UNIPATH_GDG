"""Backward-compatible import shim for the new package layout."""

from src.unipath.portal.advisor_system import AcademicAdvisorSystem


if __name__ == "__main__":
    system = AcademicAdvisorSystem()
    system.run()