# Tracking items / 追踪道具

Use the **Item catalog** page: sync the catalog, optionally scan the market, then press **Track**. Tracked items appear on the Overview, Collect, Price history and Forecast pages, and in scheduled collection.

- **Stop tracking**: the × on an item card in *Collect data*. Collected data is kept.
- **Disable temporarily / change the daily time**: *Settings → Tracked items*.
- **Files**: tracked items are stored in `data/items.json`. `items_config.json` in the repository is only the starter list copied on first launch.

Entry format:
```json
{ "name": "Iron Ore", "item_id": "id.item.iron_ore_1001", "category": "Misc", "description": "", "enabled": true, "schedule": "23:00" }
```
`item_id` is DarkerDB's id. Entries without it are looked up by name when collecting.
