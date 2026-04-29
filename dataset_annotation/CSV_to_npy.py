#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV -> .npy converter entrypoint.

This file remains the same CLI entry used by the existing workflow.
The implementation now lives in smaller modules in this folder.
"""

from converter_app import main


if __name__ == "__main__":
    main()
