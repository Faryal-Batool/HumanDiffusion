#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-start A* / Theta* trajectory annotator with clearance-aware planning
and clearance-guarded cubic-spline smoothing.

This file remains the entrypoint used by the existing workflow.
The implementation now lives in smaller modules in this folder.
"""

from annotation_app import main


if __name__ == "__main__":
    main()
