# Deploy com Docker Swarm + Traefik

## 1. Pré-requisitos na EC2

- Docker Engine instalado
- Swarm inicializado com `docker swarm init`
- DuckDNS já apontando para a instância
- portas `80`, `443` e `22` liberadas no Security Group

## 2. Variáveis no servidor

Crie um arquivo como `/opt/cfcgs-tracker/stack.env` baseado em
`deploy/swarm/stack.env.example`.

Para carregar e fazer o deploy manual:

```bash
set -a
. /opt/cfcgs-tracker/stack.env
set +a

/opt/cfcgs-tracker/deploy/swarm/deploy-traefik.sh
/opt/cfcgs-tracker/deploy/swarm/deploy-app.sh
```

## 3. Secrets e variables do GitHub

### Backend e frontend

Secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`

Variables:

- `DOCKERHUB_BACKEND_IMAGE`
- `DOCKERHUB_FRONTEND_IMAGE`

## 4. Atualização via GitHub Actions

Os workflows fazem:

1. build da imagem
2. push para o Docker Hub
3. SSH na EC2
4. `docker service update --image ...`

## 5. Serviços esperados

- `cfcgs_frontend`
- `cfcgs_backend`
- `cfcgs_postgres`
- stack `traefik`

Os nomes finais no Swarm ficam com prefixo do stack, por padrão `cfcgs_`.
