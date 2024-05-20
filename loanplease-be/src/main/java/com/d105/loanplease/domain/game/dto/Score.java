package com.d105.loanplease.domain.game.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Score {

    int score;
    String message;
    String reason;
}
