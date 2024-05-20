package com.d105.loanplease.global.cipher;

import org.springframework.security.crypto.bcrypt.BCrypt;

public class HashUtil {
    // 단방향 암호화
    // 암호화
//    private static final int LOG_ROUNDS = Integer.parseInt(System.getenv("HASH_UTIL_KEYS"));
    public static String hash(String value) {
        String customSalt = BCrypt.gensalt();
        return BCrypt.hashpw(value, customSalt);
    }

    // 검증
    public static boolean verify(String value, String hashedValue) { //매치되는지만 확인한다.
        //hashedValue 값에서 slat 값과 key 값을 분리하고
        //value 값을 분리한 솔트 값으로 해시하여 일치하는지 확인한다.
        return BCrypt.checkpw(value, hashedValue);
    }


}
