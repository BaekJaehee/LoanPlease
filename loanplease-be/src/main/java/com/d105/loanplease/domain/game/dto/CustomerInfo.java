package com.d105.loanplease.domain.game.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class CustomerInfo {
    String name;
    int age;
    String gender;
    int customerImage;
    String customerMessage;
    List<Boolean> customerMaterials;
}
