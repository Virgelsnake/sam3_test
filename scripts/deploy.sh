#!/bin/bash
# ===========================================
# SAM3 Video Segmentation - Deployment Script
# ===========================================
# Deploy backend to Hostinger VPS
# Usage: ./scripts/deploy.sh [production|staging]

set -e

ENVIRONMENT=${1:-production}
DEPLOY_USER=${DEPLOY_USER:-root}
DEPLOY_HOST=${DEPLOY_HOST:-your-vps-ip}
DEPLOY_PATH=${DEPLOY_PATH:-/opt/sam3}

echo "🚀 Deploying SAM3 Backend to $ENVIRONMENT"
echo "   Host: $DEPLOY_HOST"
echo "   Path: $DEPLOY_PATH"

# Build and push Docker image
echo ""
echo "📦 Building Docker image..."
docker build -t sam3-api:latest ./backend

# Save image to tar
echo ""
echo "💾 Saving Docker image..."
docker save sam3-api:latest | gzip > /tmp/sam3-api.tar.gz

# Copy to server
echo ""
echo "📤 Uploading to server..."
scp /tmp/sam3-api.tar.gz $DEPLOY_USER@$DEPLOY_HOST:/tmp/

# Deploy on server
echo ""
echo "🔧 Deploying on server..."
ssh $DEPLOY_USER@$DEPLOY_HOST << 'ENDSSH'
    cd /opt/sam3
    
    # Load new image
    docker load < /tmp/sam3-api.tar.gz
    
    # Stop existing containers
    docker-compose down || true
    
    # Start with new image
    docker-compose up -d
    
    # Cleanup
    rm /tmp/sam3-api.tar.gz
    docker image prune -f
    
    # Check health
    sleep 5
    curl -f http://localhost:8000/api/health || echo "⚠️ Health check failed"
ENDSSH

# Cleanup local
rm /tmp/sam3-api.tar.gz

echo ""
echo "✅ Deployment complete!"
echo "   Check status: ssh $DEPLOY_USER@$DEPLOY_HOST 'docker-compose -f $DEPLOY_PATH/docker-compose.yml ps'"
