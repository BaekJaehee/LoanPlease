package com.d105.loanplease.domain.rank.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class Rank {
    Integer rank;
    String name;
    Integer score;
    String profile;
}
