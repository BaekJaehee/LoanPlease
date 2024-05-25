
# Loan Please(론플리즈) 💰


## 목차

- [서비스 개요](#서비스-개요)
- [팀원소개](#팀원소개)
- [기술스택](#기술스택)
- [시스템 아키텍처](#시스템-아키텍처)
- [기능소개](#기능소개)
- [프로젝트 산출물](#프로젝트-산출물)
- [컨벤션](#컨벤션)


## 서비스 개요

> 신용도 예측 AI 모델을 활용한 게임형 대출 프로세스 교육 서비스
>
> Loan Please(론플리즈) 💰
> 
> 📌 2024.04.08 ~ 2024.05.24 (7주)



## 팀원소개
<div align="middle">
<table>
    <tr>
        <td height="140px" align="center"> <a href="https://github.com/SeLino98">
            <img src="https://avatars.githubusercontent.com/SeLino98" width="140px" /> <br><br> 👑 김인호  </a><br>(Back-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/nks211">
            <img src="https://avatars.githubusercontent.com/nks211" width="140px" /> <br><br> 💫 김민우 </a>  <br>(Front-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/BaekJaehee">
            <img src="https://avatars.githubusercontent.com/BaekJaehee" width="140px" /> <br><br> 🐼 백재희 </a> <br>(Front-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/kipperhr">
            <img src="https://avatars.githubusercontent.com/kipperhr" width="140px" /> <br><br> 🍎 이민수</a>  <br>(Front-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/joongwonLee">
            <img src="https://avatars.githubusercontent.com/joongwonLee" width="140px" /> <br><br> 🕶 이중원 </a><br>(Back-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/Int-TRUE">
            <img src="https://avatars.githubusercontent.com/Int-TRUE" width="140px" /> <br><br> 🍔 정수진  </a><br>(Back-End)<br></td>
        <td height="140px" align="center"> <a href="https://github.com/cnh12">
            <img src="https://avatars.githubusercontent.com/cnh12" width="140px" /> <br><br> 💚 조남현 </a> <br>(Back-End)<br></td>
    </tr>
</table>
</div>


## 기술스택

### **프론트엔드**

<div align="middle">

<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=React&logoColor=white">
<img src="https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=white">

**Language |** Javascript

**Framework |** React: 18.2.0

**IDE |** VsCode : 1.85.1

**Library |** zustand

<br>
<br>

</div>


### **백엔드**
<div align="middle">

<img src="https://img.shields.io/badge/java-3a75b0?style=for-the-badge&logo=java&logoColor=black"> <img src="https://img.shields.io/badge/spring-6DB33F?style=for-the-badge&logo=spring&logoColor=white">
<img src="https://img.shields.io/badge/mysql-4479A1?style=for-the-badge&logo=mysql&logoColor=white">
<img src="https://img.shields.io/badge/JWT-black?style=for-the-badge&logo=JSON%20web%20tokens">
<img src="https://img.shields.io/badge/-Swagger-%23Clojure?style=for-the-badge&logo=swagger&logoColor=white">
<img src="https://img.shields.io/badge/spring boot-6DB33F?style=for-the-badge&logo=springboot&logoColor=white">
<img src="https://img.shields.io/badge/spring mvc-6DB33F?style=for-the-badge&logo=spring&logoColor=white">
<img src="https://img.shields.io/badge/JPA Hibernate-59666C?style=for-the-badge&logo=Hibernate&logoColor=white">

**Language |** Oracle OpenJDK version 21.0.2

**Framework |** Spring Boot 3.2.3

**Data(RDBMS) |** MySQL 8.0.36, Redis 7.2.4


<br>
<br>

</div>


### **AI**
<div align="middle">

<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/fastapi-009688?style=for-the-badge&logo=fastapi&logoColor=white">

**Language |** Python 3.9

<br>
<br>

</div>


### **인프라**
<div align="middle">

<img src="https://img.shields.io/badge/gitlab-F05032?style=for-the-badge&logo=gitlab&logoColor=white">
<img src="https://img.shields.io/badge/AWS EC2-FF9900?style=for-the-badge&logo=amazonec2&logoColor=white">
<img src="https://img.shields.io/badge/jenkins-111111?style=for-the-badge&logo=jenkins&logoColor=white">
<img src="https://img.shields.io/badge/docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/mattermost-0058CC?style=for-the-badge&logo=mattermost&logoColor=white">
<img src="https://img.shields.io/badge/nginx-00953B?style=for-the-badge&logo=nginx&logoColor=white">

**Server |** AWS EC2 Ubuntu 20.04.6 LTS, GitLab, Jenkins(JDK 17), Docker

<br>
<br>

</div>


## 시스템 아키텍처
![image.png](./image.png)

## 기능소개
> ### ✨ 구글 회원가입/로그인
<br>
<img src = "images/2차로그인.gif" width="200">

<br>
<br>

> ### ✨ 랭킹 확인
<br>
<img src = "images/페이스아이디.gif" width="200">

- 카카오 소셜 로그인 (1차)
- 지문과 페이스 아이디를 통한 생체 로그인 (2차)

<br>
<br>

> ### ✨ 상점
<br>
<img src = "images/페이스아이디.gif" width="200">

- 카카오 소셜 로그인 (1차)
- 지문과 페이스 아이디를 통한 생체 로그인 (2차)

<br>
<br>

> ### ✨ 게임
<br>
<img src = "images/페이스아이디.gif" width="200">

- 카카오 소셜 로그인 (1차)
- 지문과 페이스 아이디를 통한 생체 로그인 (2차)

<br>
<br>

> ### ✨ 마이페이지
<br>
<img src = "images/페이스아이디.gif" width="200">

- 카카오 소셜 로그인 (1차)
- 지문과 페이스 아이디를 통한 생체 로그인 (2차)

<br>
<br>
