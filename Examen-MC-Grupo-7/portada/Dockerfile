FROM nginx:stable-alpine

COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html styles.css escena.js /usr/share/nginx/html/universo/
COPY vendor/ /usr/share/nginx/html/universo/vendor/

USER nginx
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
