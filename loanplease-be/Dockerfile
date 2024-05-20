FROM openjdk:17-jdk-alpine
VOLUME /tmp
ADD ./build/libs/loanplease-0.0.1-SNAPSHOT.jar app.jar
ENV JAVA_OPTS=""
# ENTRYPOINT ["sh", "-c", "java ${JAVA_OPTS} -jar /app.jar"]
ENTRYPOINT ["java", "-jar", "/app.jar"]

