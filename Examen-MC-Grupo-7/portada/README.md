# Portada inmersiva HTRU2

Visualización artística interactiva de un púlsar, creada con Three.js. La GPU del navegador dibuja la superficie procedural, la corona y los haces. La escena incluye líneas de campo magnético y un fondo de estrellas. La representación tiene un propósito decorativo para la exposición.

## Abrir

Publicación en https://htru2.bananacorp.dev/universo/.

Para ejecutar una copia local, abrir la terminal en esta carpeta.

```powershell
py -m http.server 8511 --bind 127.0.0.1
```

Visitar http://127.0.0.1:8511/. Las bibliotecas y la fuente acompañan la carpeta para cargar la escena sin servicios externos. El enlace al clasificador abre el dashboard público.

## Interacción

Arrastrar la escena cambia su orientación. La barra espaciadora pausa o reanuda la animación cuando el foco está en la página. La preferencia de movimiento reducido inicia la escena en pausa. La portada presenta la escena con acceso al clasificador.

El navegador necesita WebGL 2. Si falla la inicialización, la portada muestra un mensaje y conserva el enlace al clasificador.

## Despliegue en la VPS

La carpeta usa un servicio estático independiente con Nginx y la red `coolify` existente. Traefik dirige la ruta `/universo/` al contenedor.

```bash
docker compose up -d --build
docker compose ps
```

## Dependencias

Three.js 0.180.0 bajo licencia MIT. DM Sans bajo SIL Open Font License. Las licencias acompañan los archivos en `vendor/`.
