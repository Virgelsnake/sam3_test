#!/bin/bash
# ===========================================
# SAM3 Video Segmentation - VPS Setup Script
# ===========================================
# Initial setup for Hostinger VPS
# Run this once on a fresh VPS

set -e

echo "🔧 Setting up SAM3 on VPS"

# Update system
echo ""
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install Docker
echo ""
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
fi

# Install Docker Compose
echo ""
echo "📦 Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    apt-get install -y docker-compose-plugin
fi

# Create app directory
echo ""
echo "📁 Creating application directory..."
mkdir -p /opt/sam3
cd /opt/sam3

# Create .env file template
echo ""
echo "📝 Creating environment file..."
cat > /opt/sam3/.env << 'EOF'
# SAM3 Production Environment
DEBUG=false

# Supabase
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# Redis
REDIS_URL=redis://redis:6379

# Modal
MODAL_TOKEN_ID=
MODAL_TOKEN_SECRET=

# API
API_BASE_URL=http://your-domain.com:8000

# CORS
CORS_ORIGINS=["https://your-app.netlify.app"]
EOF

# Configure firewall
echo ""
echo "🔒 Configuring firewall..."
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw allow 8000/tcp # API
ufw --force enable

echo ""
echo "✅ VPS setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit /opt/sam3/.env with your credentials"
echo "2. Copy docker-compose.yml to /opt/sam3/"
echo "3. Run: cd /opt/sam3 && docker-compose up -d"
