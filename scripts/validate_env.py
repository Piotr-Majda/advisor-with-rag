import os
import sys
from typing import Dict, List, Optional

required_vars = {
    'common': [
        'OPENAI_API_KEY',
        'SERPAPI_API_KEY',
        'APP_ENV',
        'LOG_LEVEL',
        'API_GATEWAY_URL',
        'VECTOR_DB_TYPE',
    ],
    'development': [
        'DEBUG',
        'RELOAD',
    ],
    'production': [
        'ALLOWED_HOSTS',
        'CORS_ORIGINS',
        'SSL_CERT_PATH',
        'SSL_KEY_PATH',
        'SENTRY_DSN',
    ]
}

def validate_env(env_file: str) -> List[str]:
    missing_vars = []
    env_vars: Dict[str, Optional[str]] = {}
    
    # Read environment file
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    # Check common variables
    for var in required_vars['common']:
        if var not in env_vars:
            missing_vars.append(var)
    
    # Check environment-specific variables
    env_type = 'production' if 'prod' in env_file else 'development'
    for var in required_vars[env_type]:
        if var not in env_vars:
            missing_vars.append(var)
    
    return missing_vars

if __name__ == '__main__':
    env_file = sys.argv[1]
    missing = validate_env(env_file)
    if missing:
        print(f"Missing required environment variables in {env_file}:")
        for var in missing:
            print(f"- {var}")
        sys.exit(1)
    print(f"Environment file {env_file} is valid!")