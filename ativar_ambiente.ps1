# Script para ativar o ambiente virtual do projeto PraticasEmPython
Write-Host "Ativando ambiente virtual do projeto PraticasEmPython..." -ForegroundColor Green

# Ativar o ambiente virtual
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "✅ Ambiente virtual ativado!" -ForegroundColor Green
Write-Host ""
Write-Host "Para desativar, use: deactivate" -ForegroundColor Yellow
Write-Host "Para instalar novos pacotes, use: pip install nome_do_pacote" -ForegroundColor Yellow
Write-Host ""
