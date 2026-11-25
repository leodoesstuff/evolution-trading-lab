# Publish with Streamlit Community Cloud

The top-right **Deploy → Streamlit Community Cloud → Deploy now** flow requires
this source to be pushed to a GitHub repository. Use a private repository and keep
the deployed app private, because the app includes experiment controls and a
market-data connection form.

Deployment fields:

- Repository: the GitHub repository containing this source.
- Branch: `main`.
- Main file path: `app.py`.
- Python version in Advanced settings: `3.12`.
- Dependencies: the root `requirements.txt`.

Keep API keys in Streamlit's app secrets or enter them through the app's
**Data connection** form. Never put actual keys in GitHub, commit `secrets.toml`,
or include environment files in an upload. Optional app secrets use:

```toml
APCA_API_KEY_ID = "YOUR_PAPER_KEY_ID"
APCA_API_SECRET_KEY = "YOUR_PAPER_SECRET"
```

`runs/`, `practice/`, `data/`, checkpoints, and local databases are excluded from
the source repository. Publishing the source creates a separate cloud
installation; it does not move or erase experiments on the laptop. A new cloud
installation needs its own dataset, imported saved data, or synthetic demo data.
Saved local experiments continue to belong to the local installation until a
separate migration is completed.

Community Cloud can restart, throttle, or hibernate an app. A cloud app's writable
filesystem is not a permanent backup. Preserve experiment exports outside the
service before restarting or redeploying it, and use durable storage for long-term
training. Training workers run within the hosting service's lifecycle.

Official deployment instructions:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Official sharing and private-app settings:
https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app
