#!/bin/bash
gunicorn -b 0.0.0.0:$PORT Conversational_Ai:app
