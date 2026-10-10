# Scheduled collection / 定时采集

Turn it on in the **Scheduled collection** page. Each enabled tracked item is collected once a day at its time (default 23:00 local time, change it in *Settings → Tracked items*). The app must stay running; stopping the app stops the scheduler.

Command line equivalent:
```bash
python src/scheduler/start_scheduler.py --list    # show tasks built from data/items.json
python src/scheduler/start_scheduler.py --start   # run in the foreground
```
Tasks are generated from tracked items by `src/scheduler/task_config.py`; there is nothing to edit there.
