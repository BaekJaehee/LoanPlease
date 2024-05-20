package com.d105.loanplease.domain.user.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
@Entity
@Table(name = "friendship_tb")
public class Friendship {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "friendship_id")
    private Long friendshipId;

    @ManyToOne
    @JoinColumn(referencedColumnName = "user_id", name = "from_user_id")
    private User from;

    @ManyToOne
    @JoinColumn(referencedColumnName = "user_id", name = "to_user_id")
    private User to;

    private Boolean isAccess;

}
