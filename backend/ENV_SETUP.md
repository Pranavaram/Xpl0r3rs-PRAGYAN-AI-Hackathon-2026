# Setting Up Environment Variables

## Quick Setup

### Option 1: Using `.env` file (Recommended for Development)

1. **Create a `.env` file** in the project root directory:
   ```bash
   touch .env
   ```

2. **Add your Groq API key** to the `.env` file:
   ```
   GROQ_API_KEY=your-actual-api-key-here
   ```

3. **Install python-dotenv** (if not already installed):
   ```bash
   pip install python-dotenv
   ```

4. **Run the app** - it will automatically load the `.env` file:
   ```bash
   python app.py
   ```

**Note:** Make sure `.env` is in your `.gitignore` to avoid committing your API key!

### Option 2: Export in Terminal (Temporary)

#### Linux/Mac:
```bash
export GROQ_API_KEY="your-api-key-here"
python app.py
```

#### Windows (Command Prompt):
```cmd
set GROQ_API_KEY=your-api-key-here
python app.py
```

#### Windows (PowerShell):
```powershell
$env:GROQ_API_KEY="your-api-key-here"
python app.py
```

**Note:** This only works for the current terminal session. Close the terminal and you'll need to set it again.

### Option 3: Set in Your Shell Profile (Permanent for Your User)

#### Linux/Mac (bash/zsh):
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export GROQ_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

#### Windows:
Add as a system environment variable through:
- Settings → System → About → Advanced system settings → Environment Variables

### Option 4: Set in IDE (VS Code, PyCharm, etc.)

#### VS Code:
1. Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app.py",
            "env": {
                "GROQ_API_KEY": "your-api-key-here"
            }
        }
    ]
}
```

#### PyCharm:
1. Run → Edit Configurations
2. Add environment variable: `GROQ_API_KEY=your-api-key-here`

## Getting Your Groq API Key

1. Go to https://console.groq.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (you won't be able to see it again!)

## Security Best Practices

1. **Never commit `.env` files to git** - add to `.gitignore`:
   ```
   .env
   .env.local
   ```

2. **Use different keys for development and production**

3. **Rotate keys regularly**

4. **For production**, use your hosting platform's environment variable settings:
   - Heroku: `heroku config:set GROQ_API_KEY=your-key`
   - AWS: Use Systems Manager Parameter Store or Secrets Manager
   - Docker: Pass via `-e GROQ_API_KEY=your-key`
   - Kubernetes: Use Secrets

## Verify It's Set

You can verify the environment variable is set:

```bash
# Linux/Mac
echo $GROQ_API_KEY

# Windows (Command Prompt)
echo %GROQ_API_KEY%

# Windows (PowerShell)
echo $env:GROQ_API_KEY
```

## Troubleshooting

If you get `GROQ_API_KEY environment variable is not set`:

1. Check the variable is set: `echo $GROQ_API_KEY` (or equivalent for your OS)
2. Make sure you're in the same terminal session where you set it
3. If using `.env`, ensure `python-dotenv` is installed: `pip install python-dotenv`
4. Restart your IDE/terminal after setting environment variables
