#!/bin/bash
# GCP Setup Script for Polymarket Analyst
# Run this once to set up your GCP project

set -e

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================
PROJECT_ID="your-project-id"
REGION="us-central1"
SERVICE_NAME="polymarket-analyst"

# ============================================================================
# ENABLE APIS
# ============================================================================
echo "🔧 Enabling required GCP APIs..."

gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  cloudscheduler.googleapis.com \
  sqladmin.googleapis.com \
  --project=$PROJECT_ID

# ============================================================================
# CREATE SECRETS
# ============================================================================
echo "🔐 Creating secrets in Secret Manager..."

# OpenAI API Key
echo "Enter your OpenAI API key:"
read -s OPENAI_KEY
echo -n "$OPENAI_KEY" | gcloud secrets create openai-api-key \
  --data-file=- \
  --project=$PROJECT_ID \
  2>/dev/null || echo "Secret openai-api-key already exists"

# Perplexity API Key
echo "Enter your Perplexity API key:"
read -s PERPLEXITY_KEY
echo -n "$PERPLEXITY_KEY" | gcloud secrets create perplexity-api-key \
  --data-file=- \
  --project=$PROJECT_ID \
  2>/dev/null || echo "Secret perplexity-api-key already exists"

# ============================================================================
# CREATE CLOUD SQL INSTANCE (Optional - for production)
# ============================================================================
echo "Do you want to create a Cloud SQL instance? (y/n)"
read CREATE_SQL

if [ "$CREATE_SQL" = "y" ]; then
  echo "🗄️ Creating Cloud SQL instance..."
  
  gcloud sql instances create polymarket-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=$REGION \
    --project=$PROJECT_ID
  
  # Generate random password
  DB_PASSWORD=$(openssl rand -base64 24)
  
  gcloud sql users set-password postgres \
    --instance=polymarket-db \
    --password=$DB_PASSWORD \
    --project=$PROJECT_ID
  
  # Create database
  gcloud sql databases create polymarket \
    --instance=polymarket-db \
    --project=$PROJECT_ID
  
  # Store connection string in Secret Manager
  CONNECTION_NAME=$(gcloud sql instances describe polymarket-db --format='value(connectionName)' --project=$PROJECT_ID)
  DATABASE_URL="postgresql://postgres:$DB_PASSWORD@/$DB_NAME?host=/cloudsql/$CONNECTION_NAME"
  
  echo -n "$DATABASE_URL" | gcloud secrets create database-url \
    --data-file=- \
    --project=$PROJECT_ID
  
  echo "✅ Cloud SQL instance created. Connection name: $CONNECTION_NAME"
fi

# ============================================================================
# GRANT PERMISSIONS
# ============================================================================
echo "🔑 Granting permissions to Cloud Run service account..."

# Get the default compute service account
SERVICE_ACCOUNT="$PROJECT_ID@appspot.gserviceaccount.com"

# Grant Secret Manager access
gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor" \
  --project=$PROJECT_ID

gcloud secrets add-iam-policy-binding perplexity-api-key \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/secretmanager.secretAccessor" \
  --project=$PROJECT_ID

if [ "$CREATE_SQL" = "y" ]; then
  gcloud secrets add-iam-policy-binding database-url \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --project=$PROJECT_ID
  
  # Grant Cloud SQL Client role
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/cloudsql.client"
fi

# ============================================================================
# DONE
# ============================================================================
echo ""
echo "============================================"
echo "✅ GCP Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Update PROJECT_ID in cloudbuild.yaml"
echo "2. Run: gcloud builds submit --config=infrastructure/cloudbuild.yaml"
echo "3. After deployment, set up Cloud Scheduler jobs (see scheduler.yaml)"
echo ""

